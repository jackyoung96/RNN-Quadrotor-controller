// Generated by gencpp from file crazyswarm/Takeoff.msg
// DO NOT EDIT!


#ifndef CRAZYSWARM_MESSAGE_TAKEOFF_H
#define CRAZYSWARM_MESSAGE_TAKEOFF_H

#include <ros/service_traits.h>


#include <crazyswarm/TakeoffRequest.h>
#include <crazyswarm/TakeoffResponse.h>


namespace crazyswarm
{

struct Takeoff
{

typedef TakeoffRequest Request;
typedef TakeoffResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct Takeoff
} // namespace crazyswarm


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::crazyswarm::Takeoff > {
  static const char* value()
  {
    return "b665b6c83a196e4774268cc26329b159";
  }

  static const char* value(const ::crazyswarm::Takeoff&) { return value(); }
};

template<>
struct DataType< ::crazyswarm::Takeoff > {
  static const char* value()
  {
    return "crazyswarm/Takeoff";
  }

  static const char* value(const ::crazyswarm::Takeoff&) { return value(); }
};


// service_traits::MD5Sum< ::crazyswarm::TakeoffRequest> should match
// service_traits::MD5Sum< ::crazyswarm::Takeoff >
template<>
struct MD5Sum< ::crazyswarm::TakeoffRequest>
{
  static const char* value()
  {
    return MD5Sum< ::crazyswarm::Takeoff >::value();
  }
  static const char* value(const ::crazyswarm::TakeoffRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::crazyswarm::TakeoffRequest> should match
// service_traits::DataType< ::crazyswarm::Takeoff >
template<>
struct DataType< ::crazyswarm::TakeoffRequest>
{
  static const char* value()
  {
    return DataType< ::crazyswarm::Takeoff >::value();
  }
  static const char* value(const ::crazyswarm::TakeoffRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::crazyswarm::TakeoffResponse> should match
// service_traits::MD5Sum< ::crazyswarm::Takeoff >
template<>
struct MD5Sum< ::crazyswarm::TakeoffResponse>
{
  static const char* value()
  {
    return MD5Sum< ::crazyswarm::Takeoff >::value();
  }
  static const char* value(const ::crazyswarm::TakeoffResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::crazyswarm::TakeoffResponse> should match
// service_traits::DataType< ::crazyswarm::Takeoff >
template<>
struct DataType< ::crazyswarm::TakeoffResponse>
{
  static const char* value()
  {
    return DataType< ::crazyswarm::Takeoff >::value();
  }
  static const char* value(const ::crazyswarm::TakeoffResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // CRAZYSWARM_MESSAGE_TAKEOFF_H
